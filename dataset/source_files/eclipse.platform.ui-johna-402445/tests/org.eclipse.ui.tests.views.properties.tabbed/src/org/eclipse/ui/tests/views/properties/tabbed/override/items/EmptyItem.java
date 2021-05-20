/*******************************************************************************
 * Copyright (c) 2007 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.ui.tests.views.properties.tabbed.override.items;

import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.layout.FormData;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Label;
import org.eclipse.ui.views.properties.tabbed.TabbedPropertySheetWidgetFactory;

/**
 * An item for the emply selection when there is no selected element in the
 * override tests view.
 * 
 * @author Anthony Hunter
 * @since 3.4
 */
public class EmptyItem implements IOverrideTestsItem {

	private Composite composite;

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#createControls(org.eclipse.swt.widgets.Composite)
	 */
	public void createControls(Composite parent) {
		TabbedPropertySheetWidgetFactory factory = new TabbedPropertySheetWidgetFactory();
		composite = factory.createFlatFormComposite(parent);
		Label label = factory.createLabel(composite,
				"Empty Item (no selected element)"); //$NON-NLS-1$
		label.setLayoutData(new FormData());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#dispose()
	 */
	public void dispose() {
		if (composite != null && !composite.isDisposed()) {
			composite.dispose();
			composite = null;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#getComposite()
	 */
	public Composite getComposite() {
		return composite;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#getElement()
	 */
	public Class getElement() {
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#getImage()
	 */
	public Image getImage() {
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.ui.tests.views.properties.tabbed.override.items.IOverrideTestsItem#getText()
	 */
	public String getText() {
		return "Empty Item"; //$NON-NLS-1$
	}
}
