/*******************************************************************************
 * Copyright (c) 2009, 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.e4.ui.tests.reconciler;

import java.util.Collection;
import org.eclipse.e4.ui.model.application.MApplication;
import org.eclipse.e4.ui.model.application.commands.MCommand;
import org.eclipse.e4.ui.model.application.commands.impl.CommandsFactoryImpl;
import org.eclipse.e4.ui.workbench.modeling.ModelDelta;
import org.eclipse.e4.ui.workbench.modeling.ModelReconciler;

public abstract class ModelReconcilerApplicationTest extends
		ModelReconcilerTest {

	public void testApplication_Commands_Add() {
		MApplication application = createApplication();

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		MCommand command = CommandsFactoryImpl.eINSTANCE.createCommand();
		command.setCommandName("newCommand");
		application.getCommands().add(command);

		Object state = reconciler.serialize();

		application = createApplication();

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(0, application.getCommands().size());

		applyAll(deltas);

		assertEquals(1, application.getCommands().size());

		command = application.getCommands().get(0);
		assertEquals("newCommand", command.getCommandName());
	}

	public void testApplication_Commands_Remove() {
		MApplication application = createApplication();

		MCommand command = CommandsFactoryImpl.eINSTANCE.createCommand();
		application.getCommands().add(command);

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		application.getCommands().remove(0);

		Object state = reconciler.serialize();

		application = createApplication();

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getCommands().size());
		command = application.getCommands().get(0);

		applyAll(deltas);

		assertEquals(0, application.getCommands().size());
	}
}
